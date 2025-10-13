from __future__ import annotations

import torch

import helion
import helion.language as hl

@helion.kernel(
    use_default_config=True,
    static_shapes=True,
)
def unbind_bug(q: torch.Tensor) -> torch.Tensor:
    _, _, M, D = q.shape
    D = hl.specialize(D)
    M = hl.specialize(M)
    q = q.reshape(-1, D)
    total_rows = q.shape[0]
    block_m = hl.register_block_size(M)
    result = hl.zeros([total_rows, D])
    for tile_m in hl.tile(total_rows, block_size=block_m):
        acc = hl.zeros([tile_m, D])

        for tile_n in hl.tile(M, block_size=block_m):
            reshaped = acc.reshape(tile_m, 2, D // 2)
            acc0, acc1 = torch.unbind(reshaped, dim=1)
            acc = torch.stack((acc0, acc1), dim=1).reshape(tile_m, D)
            acc = acc + 0

        result[tile_m, :] = acc

    return result


def main() -> None:
    q = torch.randn(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    expected = q.reshape(-1, q.shape[-1])
    out = unbind_bug(q)
    torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    main()
