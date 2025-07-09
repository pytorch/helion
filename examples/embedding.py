from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        block_sizes=[512, 32], loop_order=[0, 1], num_warps=8, indexing="block_ptr"
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


def embedding_tritonbench(
    V: int, D: int, inp: torch.Tensor, shared_weight: torch.Tensor
) -> torch.Tensor:
    """Wrapper for tritonbench that matches its interface."""
    return embedding(inp, shared_weight)


def main() -> None:
    num_embeddings, embedding_dim = 16, 64
    x = torch.randint(0, num_embeddings, [256, 32], device="cuda", dtype=torch.int32)
    weight = torch.randn([num_embeddings, embedding_dim], device="cuda")
    run_example(
        embedding, torch.nn.functional.embedding, (x, weight), atol=0.0, rtol=0.0
    )


if __name__ == "__main__":
    main()
