"""An online softmax's row statistics: two passes over the same lanes.

Shared by the lane-load stash's codegen tests
(``test/test_cute_stash_lane_loads.py``) and its GPU twin
(``test/test_cute_lane_loop_reduce_two_pass.py``): over an ``hl.tile`` loop
the block max and the sum of ``exp(x - max)`` are two passes over the
thread's lanes, the second reading the elements the first loaded.
"""

from __future__ import annotations

import torch

import helion.language as hl


def online_softmax_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's max and sum of ``exp(x - max)``, one block at a time."""
    rows, n = x.shape
    row_max = torch.empty([rows], dtype=torch.float32, device=x.device)
    row_sum = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        run_max = hl.full([tile_r], float("-inf"), dtype=torch.float32)
        run_sum = hl.zeros([tile_r], dtype=torch.float32)
        for tile_v in hl.tile(n):
            v = x[tile_r, tile_v].to(torch.float32)
            new_max = torch.maximum(run_max, torch.amax(v, dim=-1))
            run_sum = run_sum * torch.exp(run_max - new_max) + torch.sum(
                torch.exp(v - new_max[:, None]), dim=-1
            )
            run_max = new_max
        row_max[tile_r] = run_max
        row_sum[tile_r] = run_sum
    return row_max, row_sum
