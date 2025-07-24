from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel
def layer_norm_fwd(
    x: torch.Tensor,
    nomralized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {m}"
    assert len(nomralized_shape) == 1, (
        "Helion layer norm only supports 1D layer norm currently"
    )
    assert nomralized_shape[0] == n, (
        f"normalized shape mismatch {nomralized_shape[0]} != {n}"
    )

    out = torch.empty([m, n], dtype=torch.float16, device=x.device)

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)

        var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)

        normalized = (acc - mean) * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))

        out[tile_m, :] = acc
    return out


def main() -> None:
    batch_size = 32
    dim = 64
    device = "cuda"

    x = torch.randn([batch_size, dim], device=device, dtype=torch.float16)
    weight = torch.randn([dim], device=device, dtype=torch.float16)
    bias = torch.randn([dim], device=device, dtype=torch.float16)
    eps = 1e-4

    run_example(
        layer_norm_fwd,
        torch.nn.functional.layer_norm,
        (x, [dim], weight, bias, eps),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    main()
