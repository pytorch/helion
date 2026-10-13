"""Single-call T128 RetNet prefill with a fixed-decay distance formula.

Inputs are BF16 token-major q/k/v, a read-only FP32 K-major initial state,
and the original FP32 per-chunk prefix (H, 2, 64). The prefix must come from
log(1 - 2**(-5-head)), repeated for 64 tokens and cumulatively summed in FP32.
This explicit fixed-decay entry uses prefix[head, 0, 0] and FP32 distance
multiplication; it does not implement arbitrary learned or token-varying gates.

The BF16 output is newly allocated; final_state is the supplied FP32 destination.
Q/K dimension and sequence length are 128, chunk_size is 64. Query rows use a
fixed algorithmic tile of 32; value tiling and schedules remain ordinary compiler
configuration choices. There is no input-shape-based configuration dispatch.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

RCP_LN2 = 1.4426950408889634


@helion.kernel(static_shapes=True, backend="cute", fast_math=False)
def retention_prefill_fixed_decay(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    scale: float,
    chunk_size: hl.constexpr = 64,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, D = q.shape
    D = hl.specialize(D)
    DV = v.size(3)
    assert tokens == D == 128 and chunk_size == 64 and (prefix.size(1) == 2)
    output = torch.empty_like(v)
    for tile_batch, tile_head, tile_row, tile_dv in hl.tile(
        [batch, heads, D, DV], block_size=[1, 1, 32, None]
    ):
        sequence = tile_batch.id
        head = tile_head.id
        state = initial_state[sequence, head, tile_row, tile_dv].float()
        for once in hl.grid(1):
            channel = hl.arange(D)
            query_index = tile_row.index
            key_index = hl.arange(128) + once * 128
            query = hl.load(
                q,
                [sequence, query_index, head, channel],
                extra_mask=(query_index < tokens)[:, None],
            )
            key = hl.load(
                k,
                [sequence, key_index, head, channel],
                extra_mask=(key_index < tokens)[:, None],
            )
            value = hl.load(
                v,
                [sequence, key_index, head, tile_dv],
                extra_mask=(key_index < tokens)[:, None],
            )
            entering = hl.load(
                initial_state,
                [sequence, head, channel, tile_dv],
                extra_mask=(channel < D)[:, None],
            ).to(k.dtype)
            lg = prefix[head, 0, 0].float()
            cross = hl.dot(query, entering)
            gram = hl.dot(query, key.T)
            attention = torch.where(
                query_index[:, None] >= key_index[None, :],
                gram
                * torch.exp2(
                    (query_index[:, None] - key_index[None, :]).float() * lg * RCP_LN2
                ),
                0.0,
            )
            cross = (
                cross * torch.exp2((query_index + 1).float() * lg * RCP_LN2)[:, None]
            )
            result = hl.dot(attention.to(v.dtype), value, acc=cross)
            hl.store(
                output,
                [sequence, query_index, head, tile_dv],
                (result * scale).to(output.dtype),
                extra_mask=(query_index < tokens)[:, None],
            )
            state_key = hl.load(
                k,
                [sequence, key_index, head, tile_row],
                extra_mask=(key_index < tokens)[:, None],
            )
            weighted = (
                value.float()
                * torch.exp2((127 - key_index).float() * lg * RCP_LN2)[:, None]
            ).to(v.dtype)
            state = hl.dot(
                state_key.T, weighted, acc=state * torch.exp2(128.0 * lg * RCP_LN2)
            )
        final_state[sequence, head, tile_row, tile_dv] = state
    return (output, final_state)


def main() -> None:
    device = torch.device("cuda")
    q = torch.randn((1, 128, 8, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    initial = torch.zeros((1, 8, 128, 128), device=device, dtype=torch.float32)
    final = torch.empty_like(initial)
    head = torch.arange(8, device=device, dtype=torch.float32)
    two = torch.tensor(2.0, device=device, dtype=torch.float32)
    log_gamma = (1.0 - two.pow(-5.0 - head)).log()
    prefix = log_gamma[:, None, None].expand(8, 2, 64).cumsum(-1)
    output, final = retention_prefill_fixed_decay(
        q, k, v, prefix, initial, final, 128**-0.5
    )
    torch.cuda.synchronize()
    print(output.shape, final.shape)


if __name__ == "__main__":
    main()
