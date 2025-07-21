from __future__ import annotations

from typing import Any

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(static_shapes=True, use_default_config=True)
def layer_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {m}"
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)

    eps = 1e-5

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(
            torch.float32
        )  # TODO (PaulZhang12): Eliminate this cast, currently necessary

        var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)

        normalized = (acc - mean) * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))

        out[tile_m, :] = acc
    return out


def layer_norm_torch_callable(
    dims: list[int],
) -> Any:  # noqa: ANN401
    return lambda x, weight, bias, eps: torch.nn.functional.layer_norm(
        x, dims, weight, bias, eps
    )


def main() -> None:
    batch_size = 32
    dim = 64
    device = "cuda"
    eps = 1e-3

    x = torch.randn([batch_size, dim], device=device, dtype=torch.float16)
    weight = torch.randn([dim], device=device, dtype=torch.float16)
    bias = torch.randn([dim], device=device, dtype=torch.float16)

    run_example(
        layer_norm_fwd,
        layer_norm_torch_callable,
        (x, weight, bias, eps),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    main()
