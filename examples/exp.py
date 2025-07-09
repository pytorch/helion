from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def exp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.exp(x[tile])
    return out


def exp_tritonbench(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Wrapper for tritonbench that returns output in expected format."""
    return {"output": exp(x)}


def check(n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    run_example(exp, torch.exp, (x,))


def main() -> None:
    check(1024 * 1024)


if __name__ == "__main__":
    main()
