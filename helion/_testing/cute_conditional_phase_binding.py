"""Exact previously used CPU binding inputs for the authored PR source."""

from __future__ import annotations

import torch


def phase_fixture(batch: int = 3, buckets: int = 2) -> tuple[object, ...]:
    generator = torch.Generator().manual_seed(461)
    storage = torch.randn(batch * 18432 + 8, generator=generator).bfloat16()
    q = storage.as_strided((batch, 4, 16, 128), (18432, 4608, 128, 1), 8)
    packed = []
    for _ in range(2):
        raw = torch.randint(-8, 8, (7 * 33792,), generator=generator, dtype=torch.int8)
        data = raw.as_strided((7, 128, 2, 128), (33792, 256, 128, 1))
        scale = raw.view(torch.uint8).as_strided(
            (7, 128, 2, 4), (33792, 4, 512, 1), 32768
        )
        scale.fill_(127)
        packed.append((data, scale))
    table = (torch.arange(batch * 32).reshape(batch, 32) % 7).to(torch.int32)
    lengths = torch.tensor(
        ([0, 2051, 128] * ((batch + 2) // 3))[:batch], dtype=torch.int32
    )
    output = torch.empty((batch, 4, 16, 128), dtype=torch.bfloat16)
    lse = torch.empty((batch, 4, 16), dtype=torch.float32)
    return (
        q,
        packed[0][0],
        packed[1][0],
        packed[0][1],
        packed[1][1],
        table,
        lengths,
        output,
        lse,
        buckets,
        18,
        2047,
    )
