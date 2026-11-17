"""Inputs, arguments and the FP32 reference of the BF16 verify-layer megakernels.

``llama32_1b_verify_layer_bf16`` and ``qwen3_verify_layer_bf16`` take the same
tensors and differ in sizes and in Qwen3's per-head q/k RMSNorm (``Layer``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


# The kernels' results checked, by position in the returned tuple.
OUTPUT = 0
ATTENTION = 5
RESIDUAL = 10


@dataclass(frozen=True)
class Layer:
    hidden: int
    intermediate: int
    q_heads: int
    kv_heads: int
    head_dim: int
    context: int
    cache_block: int
    attention_splits: int
    group: int
    eps: float
    rope_theta: float
    # Qwen3: an RMSNorm over each q and k head (weights q_weight / k_weight).
    qk_norm: bool


def rope_cache(layer: Layer, device: torch.device) -> torch.Tensor:
    """Neox cos | sin halves per position, BF16 ([context, head_dim])."""
    d = layer.head_dim
    inv = 1.0 / (layer.rope_theta ** (torch.arange(0, d, 2, device=device).float() / d))
    freqs = torch.arange(layer.context, device=device).float()[:, None] * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)


def make_inputs(
    layer: Layer,
    seed: int,
    context_len: int,
    tokens: int,
    init: bool = True,
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """One sequence: ``context_len`` cached rows, then ``tokens`` new tokens;
    ``init=False`` leaves the floating tensors uninitialized (code generation
    needs only their shapes). The cache rows past the sequence on its last
    page are NaN: stale rows must not reach the outputs."""
    if not (context_len >= 1 and context_len + tokens <= layer.context):
        raise ValueError("context out of range")
    torch.manual_seed(seed)
    randn = torch.randn if init else torch.empty
    hidden, d = layer.hidden, layer.head_dim
    qkv_width = (layer.q_heads + 2 * layer.kv_heads) * d
    logical_blocks = math.ceil(layer.context / layer.cache_block)
    physical_blocks = math.ceil(logical_blocks * 1.25)
    block_table = (
        torch.randperm(physical_blocks, device=device, dtype=torch.int64)[
            :logical_blocks
        ]
        .to(torch.int32)
        .view(1, logical_blocks)
    )
    positions = torch.arange(
        context_len, context_len + tokens, device=device, dtype=torch.int64
    )
    slots = (
        block_table[0, positions // layer.cache_block].to(torch.int64)
        * layer.cache_block
        + positions % layer.cache_block
    )

    def weight(rows: int, cols: int) -> torch.Tensor:
        return randn((rows, cols), device=device, dtype=torch.bfloat16) * (
            1.0 / math.sqrt(cols)
        )

    def norm_weight(size: int) -> torch.Tensor:
        return randn(size, device=device, dtype=torch.bfloat16) * 0.1 + 1.0

    def rows() -> torch.Tensor:
        return torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)

    tensors = {
        "hidden_states": randn((tokens, hidden), device=device, dtype=torch.bfloat16),
        "residual": randn((tokens, hidden), device=device, dtype=torch.bfloat16),
        "pre_weight": norm_weight(hidden),
        "pre_normed": rows(),
        "qkv_weight": weight(qkv_width, hidden),
        **(
            {"q_weight": norm_weight(d), "k_weight": norm_weight(d)}
            if layer.qk_norm
            else {}
        ),
        "cos_sin": rope_cache(layer, torch.device(device)),
        "position": positions,
        "kv_cache": randn(
            (physical_blocks, layer.cache_block, layer.kv_heads, 2 * d),
            device=device,
            dtype=torch.bfloat16,
        ),
        "block_table": block_table,
        "slot_mapping": slots,
        "o_weight": weight(hidden, hidden),
        "post_weight": norm_weight(hidden),
        "ffn_normed": rows(),
        "w13": weight(2 * layer.intermediate, hidden),
        "w2": weight(hidden, layer.intermediate),
        # The sequence length, new tokens included.
        "context_lens": torch.full(
            (1,), context_len + tokens, device=device, dtype=torch.int64
        ),
    }
    end = context_len + tokens
    if init and end % layer.cache_block:
        last = block_table[0, end // layer.cache_block]
        tensors["kv_cache"][last, end % layer.cache_block :] = float("nan")
    return tensors


def cache_rows(layer: Layer, tensors: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """The KV rows this step writes, one [kv_heads, 2 * head_dim] view per token."""
    rows = []
    for slot in tensors["slot_mapping"].tolist():
        block, offset = divmod(int(slot), layer.cache_block)
        rows.append(tensors["kv_cache"][block, offset])
    return rows


def kernel_args(layer: Layer, tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    norms = ("q_weight", "k_weight") if layer.qk_norm else ()
    names = (
        "hidden_states",
        "residual",
        "pre_weight",
        "pre_normed",
        "qkv_weight",
        *norms,
        "cos_sin",
        "position",
        "kv_cache",
        "block_table",
        "slot_mapping",
        "o_weight",
        "post_weight",
        "ffn_normed",
        "w13",
        "w2",
        "context_lens",
    )
    return (
        *(tensors[name] for name in names),
        layer.hidden,
        layer.q_heads,
        layer.kv_heads,
        layer.head_dim,
        layer.context,
        layer.cache_block,
        layer.attention_splits,
        layer.group,
        layer.eps,
    )


def reference(
    layer: Layer, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """FP32 PyTorch layer (BF16 at the kernel's stored intermediates)."""
    f = torch.float32
    d, eps = layer.head_dim, layer.eps
    q_heads, kv_heads = layer.q_heads, layer.kv_heads

    def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.to(f)

    tokens = tensors["hidden_states"].size(0)
    residual = (tensors["hidden_states"].to(f) + tensors["residual"].to(f)).to(
        torch.bfloat16
    )
    x = rms(residual.to(f), tensors["pre_weight"])
    qkv = (x.to(torch.bfloat16).to(f) @ tensors["qkv_weight"].to(f).T).to(
        torch.bfloat16
    )
    q, k, v = qkv.to(f).split([q_heads * d, kv_heads * d, kv_heads * d], -1)
    q = q.view(tokens, q_heads, d)
    k = k.view(tokens, kv_heads, d)
    v = v.view(tokens, kv_heads, d)
    if layer.qk_norm:
        q, k = rms(q, tensors["q_weight"]), rms(k, tensors["k_weight"])
    cs = tensors["cos_sin"][tensors["position"]].to(f)
    cos, sin = cs[:, None, : d // 2], cs[:, None, d // 2 :]

    def rope(t: torch.Tensor) -> torch.Tensor:
        lo, hi = t[..., : d // 2], t[..., d // 2 :]
        return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], -1)

    q, k = rope(q), rope(k)
    cached = int(tensors["context_lens"][0]) - tokens
    n = torch.arange(cached, device=q.device)
    blocks = tensors["block_table"][0, n // layer.cache_block].to(torch.int64)
    rows = tensors["kv_cache"][blocks, n % layer.cache_block].to(f)  # [cached, KV, 2D]
    keys = torch.cat([rows[..., :d], k], 0)
    values = torch.cat([rows[..., d:], v], 0)
    keys = keys.repeat_interleave(q_heads // kv_heads, 1)
    values = values.repeat_interleave(q_heads // kv_heads, 1)
    scores = torch.einsum("thd,nhd->htn", q, keys) / math.sqrt(d)
    visible = torch.arange(cached + tokens, device=q.device)[None, :] <= (
        cached + torch.arange(tokens, device=q.device)[:, None]
    )
    scores = scores.masked_fill(~visible[None], float("-inf"))
    attention = torch.einsum("htn,nhd->thd", scores.softmax(-1), values)
    attention = attention.reshape(tokens, -1).to(torch.bfloat16)
    o = (attention.to(f) @ tensors["o_weight"].to(f).T).to(torch.bfloat16)
    residual2 = (o.to(f) + residual.to(f)).to(torch.bfloat16)
    y = rms(residual2.to(f), tensors["post_weight"])
    gate_up = y.to(torch.bfloat16).to(f) @ tensors["w13"].to(f).T
    gate, up = gate_up.chunk(2, -1)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    out = (act.to(f) @ tensors["w2"].to(f).T).to(torch.bfloat16)
    return {
        "output": out,
        "attention": attention,
        "residual": residual2,
        "k_new": k,
        "v_new": v,
    }


def correctness_check(
    layer: Layer,
    kernel: Callable[..., tuple[torch.Tensor, ...]],
    tokens: int,
    context_lens: tuple[int, ...],
) -> None:
    """Check ``kernel``'s outputs and the KV rows it writes to the cache against
    ``reference``, at each of ``context_lens``."""
    d = layer.head_dim
    for context_len in context_lens:
        tensors = make_inputs(layer, 0, context_len, tokens)
        # Before the kernel writes the residual and the cache rows in place.
        expected = reference(layer, tensors)
        result = kernel(*kernel_args(layer, tensors))
        rows = torch.stack(cache_rows(layer, tensors))
        for output, got in (
            ("output", result[OUTPUT]),
            ("attention", result[ATTENTION].reshape(tokens, -1)),
            ("residual", result[RESIDUAL]),
            ("k_new", rows[..., :d]),
            ("v_new", rows[..., d:]),
        ):
            torch.testing.assert_close(
                got.float(),
                expected[output].float(),
                atol=0.05,
                rtol=0.02,
                msg=lambda m, o=output, n=context_len: f"{o}, context {n}: {m}",
            )
            # And relative to the output's norm: over a long context attention
            # values shrink toward the elementwise tolerance (the kernels are
            # within 7e-3; a stale last cache block gives 0.13).
            error = (got.float() - expected[output].float()).norm()
            relative = float(error / expected[output].float().norm())
            if relative > 2e-2:
                raise AssertionError(
                    f"{output}, context {context_len}: relative error {relative:.2e}"
                )
