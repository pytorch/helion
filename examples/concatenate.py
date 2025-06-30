from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(
    config=helion.Config(block_size=[4, 1024], loop_order=[1, 0], num_warps=2)
)
def concat2d_dim1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.size(0) == y.size(0)
    out = torch.empty(
        [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
    )
    for tile0, tile1 in hl.tile(out.size()):
        # Most masking is automatic in helion, but tile1 spans both x and y we need to do some manual masking
        x_part = hl.load(
            x, [tile0, tile1], extra_mask=(tile1.index < x.size(1))[None, :]
        )
        y_part = hl.load(
            y,
            [tile0, tile1.index - x.size(1)],
            extra_mask=(tile1.index >= x.size(1))[None, :],
        )
        out[tile0, tile1] = torch.where(
            (tile1.index < x.size(1))[None, :], x_part, y_part
        )
    return out


def main() -> None:
    x = torch.randn([1500, 400], device="cuda")
    y = torch.randn([1500, 600], device="cuda")
    run_example(concat2d_dim1, lambda x, y: torch.cat([x, y], dim=1), (x, y))


if __name__ == "__main__":
    main()
