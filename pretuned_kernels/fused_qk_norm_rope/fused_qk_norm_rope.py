"""Fused QK RMSNorm + RoPE over a packed QKV tensor.

For a packed ``qkv`` of shape ``[num_tokens, (q_heads + 2*kv_heads) * head_dim]``,
this applies per-head RMSNorm (with separate ``q_weight`` / ``k_weight``) to the Q
and K heads, then rotary position embedding (RoPE) to those same heads, writing the
results back into ``qkv`` in place. The V heads are left untouched. Ported from
vLLM's Helion ``fused_qk_norm_rope`` kernel (the checked-in heuristic is converted
from vLLM's per-hardware config JSON).
"""

from __future__ import annotations

import torch

import helion
import helion.experimental
import helion.language as hl

# Optional vLLM baseline: the production composition this is benchmarked against.
# The pretuned test env has only torch + helion (guarded import); the nightly
# benchmark workflow installs vLLM, so main() then compares against the real
# vLLM ops (rms_norm + RotaryEmbedding.forward_static). Both pieces must import
# for the vLLM baseline to be used.
try:
    from vllm import ir as _vllm_ir
    from vllm.model_executor.layers.rotary_embedding import (
        RotaryEmbedding as _VllmRotaryEmbedding,
    )

    _HAS_VLLM = hasattr(_vllm_ir.ops, "rms_norm") and hasattr(
        _VllmRotaryEmbedding, "forward_static"
    )
    if _HAS_VLLM:
        # Match vLLM's CUDA eager default (see cuda.py get_default_ir_op_priority)
        # so the baseline dispatches rms_norm to the compiled vllm_c kernel
        # instead of the pure-PyTorch "native" fallback -- otherwise the vLLM
        # baseline is artificially slow (and logs a "Priority not set" warning).
        # import_ir_kernels() registers the platform's impls (e.g. vllm_c) that
        # set_default requires; it's lazily imported, like vLLM's own KernelConfig.
        from vllm.platforms import current_platform

        current_platform.import_ir_kernels()
        _vllm_ir.ops.rms_norm.set_default(["vllm_c", "native"])
except ImportError:
    _HAS_VLLM = False


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def fused_qk_norm_rope(
    qkv: torch.Tensor,  # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor,  # [num_tokens],
    forced_token_heads_per_warp: int = -1,  # dummy
) -> None:
    assert qkv.ndim == 2
    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v
    assert qkv.shape[1] == total_heads * head_dim
    hl.specialize(qkv.shape[1])

    assert cos_sin_cache.ndim == 2
    max_position, rotary_dim = cos_sin_cache.shape
    hl.specialize(max_position)
    hl.specialize(rotary_dim)
    assert rotary_dim % 2 == 0
    assert rotary_dim <= head_dim
    embed_dim = rotary_dim // 2

    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)

    assert position_ids.ndim == 1 and position_ids.shape[0] == num_tokens
    hl.specialize(position_ids.shape[0])

    assert q_weight.ndim == 1 and q_weight.shape[0] == head_dim
    hl.specialize(q_weight.shape[0])
    assert k_weight.ndim == 1 and k_weight.shape[0] == head_dim
    hl.specialize(k_weight.shape[0])

    assert qkv.dtype == q_weight.dtype and q_weight.dtype == k_weight.dtype
    assert position_ids.dtype == torch.int64

    assert qkv.is_contiguous()
    assert position_ids.is_contiguous()
    assert q_weight.is_contiguous()
    assert k_weight.is_contiguous()
    assert cos_sin_cache.is_contiguous()

    qk_heads = num_heads_q + num_heads_k

    qkv = qkv.view(num_tokens, -1, head_dim)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    ):
        x_blk = qkv[tile_m, tile_gn, tile_n].to(dtype=torch.float32)

        rms = x_blk.pow(2).sum(dim=-1)
        rms = torch.rsqrt(rms * (1.0 / head_dim) + eps)

        use_q_weight = (tile_gn.index < num_heads_q)[None, :, None]
        w_blk = torch.where(
            use_q_weight, q_weight[None, None, tile_n], k_weight[None, None, tile_n]
        )

        x_blk = (x_blk * rms[:, :, None]).to(qkv.dtype) * w_blk

        qkv[tile_m, tile_gn, tile_n] = x_blk

        pos_id = position_ids[tile_m]
        cos_blk = cos_sin_cache[pos_id, hl.arange(embed_dim)]
        sin_blk = cos_sin_cache[pos_id, hl.arange(embed_dim) + embed_dim]

        if is_neox:
            x1_offset = hl.arange(embed_dim)
            x2_offset = x1_offset + embed_dim
        else:
            x1_offset = hl.arange(embed_dim) * 2
            x2_offset = x1_offset + 1

        x1_blk = qkv[tile_m, tile_gn, x1_offset]
        x2_blk = qkv[tile_m, tile_gn, x2_offset]

        o1_blk = x1_blk * cos_blk[:, None, :] - x2_blk * sin_blk[:, None, :]
        o2_blk = x2_blk * cos_blk[:, None, :] + x1_blk * sin_blk[:, None, :]

        qkv[tile_m, tile_gn, x1_offset] = o1_blk
        qkv[tile_m, tile_gn, x2_offset] = o2_blk


def _fused_qk_norm_rope_torch(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,  # dummy
) -> None:
    """Torch-native reference: per-head RMSNorm on Q/K then RoPE, written in place."""
    num_tokens = qkv.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    embed_dim = rotary_dim // 2
    qk_heads = num_heads_q + num_heads_k

    qkv_view = qkv.view(num_tokens, -1, head_dim)

    # Per-head RMSNorm over the Q and K heads (V untouched).
    x = qkv_view[:, :qk_heads, :].to(torch.float32)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    head_idx = torch.arange(qk_heads, device=qkv.device)
    w = torch.where(
        (head_idx < num_heads_q)[None, :, None],
        q_weight[None, None, :],
        k_weight[None, None, :],
    )
    x = (x * rms).to(qkv.dtype) * w
    qkv_view[:, :qk_heads, :] = x

    # RoPE over the Q and K heads.
    pos = position_ids
    cos = cos_sin_cache[pos, :embed_dim]  # [num_tokens, embed_dim]
    sin = cos_sin_cache[pos, embed_dim:rotary_dim]

    if is_neox:
        x1_off = torch.arange(embed_dim, device=qkv.device)
        x2_off = x1_off + embed_dim
    else:
        x1_off = torch.arange(embed_dim, device=qkv.device) * 2
        x2_off = x1_off + 1

    heads = qkv_view[:, :qk_heads, :]
    x1 = heads[:, :, x1_off]
    x2 = heads[:, :, x2_off]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    o1 = x1 * cos_b - x2 * sin_b
    o2 = x2 * cos_b + x1 * sin_b
    heads[:, :, x1_off] = o1
    heads[:, :, x2_off] = o2
    qkv_view[:, :qk_heads, :] = heads


def _fused_qk_norm_rope_vllm(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,  # dummy
) -> None:
    """vLLM baseline: a Python composition of vLLM's rms_norm + RoPE ops.

    Replicates the production baseline (vLLM PR 44010): per-head RMSNorm on the Q
    and K heads (via ``vllm.ir.ops.rms_norm``) followed by rotary embedding (via
    ``RotaryEmbedding.forward_static``), written back into ``qkv`` in place.
    """
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim

    q, k, _v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = _vllm_ir.ops.rms_norm(q_by_head, q_weight, eps)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = _vllm_ir.ops.rms_norm(k_by_head, k_weight, eps)
    k = k_by_head.view(k.shape)

    q, k = _VllmRotaryEmbedding.forward_static(
        position_ids, q, k, head_dim, cos_sin_cache.shape[1], cos_sin_cache, is_neox
    )
    qkv[:, :q_size].copy_(q)
    qkv[:, q_size : q_size + kv_size].copy_(k)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _fused_qk_norm_rope_torch),
        ("torch_compile", torch.compile(_fused_qk_norm_rope_torch)),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _fused_qk_norm_rope_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def _compute_cos_sin_cache(
    max_position_embeddings: int,
    rotary_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)
    )
    t = torch.arange(max_position_embeddings, device=device, dtype=dtype)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)


def _make_inputs(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, ...]:
    head_dim = 128
    in_dtype = torch.bfloat16
    eps = 1e-6
    is_neox = True
    rotary_dim = head_dim
    total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.empty(num_tokens, total_dim, dtype=in_dtype, device="cuda").uniform_(
        -0.1, 0.1
    )
    positions = torch.arange(num_tokens, dtype=torch.long, device="cuda")
    q_weight = torch.empty(head_dim, dtype=in_dtype, device="cuda").uniform_(0.8, 1.2)
    k_weight = torch.empty(head_dim, dtype=in_dtype, device="cuda").uniform_(0.8, 1.2)
    cos_sin_cache = _compute_cos_sin_cache(40960, rotary_dim).to(in_dtype)
    return (
        qkv,
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        positions,
    )


def _bench_shapes() -> list[tuple[int, int, int]]:
    """The (num_tokens, q_heads, kv_heads) shapes main() benchmarks."""
    num_tokens_list = [1, 8, 32, 128, 512, 2048, 8192]
    num_heads_pair = [(16, 8), (32, 8), (64, 8)]
    return [(t, qh, kvh) for (qh, kvh) in num_heads_pair for t in num_tokens_list]


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    for num_tokens, q_heads, kv_heads in _bench_shapes():
        args = _make_inputs(num_tokens, q_heads, kv_heads)
        qkv = args[0]
        qkv_helion = qkv.clone()
        qkv_torch = qkv.clone()
        fused_qk_norm_rope(qkv_helion, *args[1:])
        _fused_qk_norm_rope_torch(qkv_torch, *args[1:])
        torch.testing.assert_close(qkv_helion, qkv_torch, rtol=2e-2, atol=2e-2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    shapes = _bench_shapes()
    baselines = _baselines()

    def make_calls(shape: tuple) -> tuple:
        num_tokens, q_heads, kv_heads = shape
        args = _make_inputs(num_tokens, q_heads, kv_heads)
        rest = args[1:]
        qkv = args[0]
        qkv_helion = qkv.clone()

        def helion_call() -> None:
            fused_qk_norm_rope(qkv_helion, *rest)

        # Each baseline gets its own qkv clone (the op mutates it in place).
        base_calls = []
        for name, fn in baselines:
            qkv_base = qkv.clone()
            base_calls.append((name, lambda fn=fn, q=qkv_base: fn(q, *rest)))
        return (
            helion_call,
            base_calls,
            f"{num_tokens:>7d}  {q_heads:>4d}  {kv_heads:>5d}",
        )

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'tokens':>7s}  {'q_h':>4s}  {'kv_h':>5s}",
    )


if __name__ == "__main__":
    # Verify numerics across every benchmarked shape before timing.
    correctness_check()
    main()
