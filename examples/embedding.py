from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        block_size=[512, 32], loop_order=[0, 1], num_warps=8, indexing="block_ptr"
    )
)
def embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(-1)  # collapse x into a single dimension
    _, embedding_dim = weight.size()
    out = torch.empty(
        [x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device
    )
    for tile_b, tile_e in hl.tile([x_flat.size(0), embedding_dim]):
        out[tile_b, tile_e] = weight[x_flat[tile_b], tile_e]
    # restore the original shape
    return out.view(*x.size(), embedding_dim)


def check() -> None:
    from triton.testing import do_bench

    num_embeddings, embedding_dim = 16, 64
    x = torch.randint(0, num_embeddings, [256, 32], device="cuda", dtype=torch.int32)
    weight = torch.randn([num_embeddings, embedding_dim], device="cuda")
    result = embedding(x, weight)
    torch.testing.assert_close(result, torch.nn.functional.embedding(x, weight))
    sec = do_bench(lambda: embedding(x, weight))
    baseline_sec = do_bench(lambda: torch.nn.functional.embedding(x, weight))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


if __name__ == "__main__":
    check()
