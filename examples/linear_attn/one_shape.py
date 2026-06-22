"""Run a single shape's forward once, for ncu profiling. Usage: one_shape.py B T H D"""

from __future__ import annotations

import math
import sys

import torch

from examples.linear_attn.chunk import chunk_linear_attn


def main() -> None:
    b, t, h, d = (int(x) for x in sys.argv[1:5])
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(2):
            o = chunk_linear_attn(q, k, v, scale=scale)
        torch.cuda.synchronize()
    print("ok", tuple(o.shape))


if __name__ == "__main__":
    main()
