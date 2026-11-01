"""grouped_gemm: ``cat([A_g @ B for g in range(4)])`` with one shared B.

The reference runs in float32 on the GPU: a float64 host GEMM of the largest
row (4.4 TFLOP) would take minutes, and float32 is exact far below bf16 ulp.

Helion's side is a kernel of this module, not examples/grouped_gemm.py's,
the one other benchmarks run.  The example's persistent kernel hand-rolls its
schedule: ``hl.grid(get_num_sm(device))`` workers, each taking every
num_workers-th tile of every group, so the config can't choose the launch.
Here the same tiles are one ``hl.grid`` of (group, M tile, N tile) work
items, which the config's ``pid_type`` runs flat or as persistent workers and
``num_sm_multiplier`` sizes.  The tile body is the example's.  It is ordinary
Helion with nothing backend-specific, and the example itself is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._grouped_gemm_grid import grouped_gemm_jagged_grid
from helion._testing import import_example

_reference_grouped_gemm = import_example("grouped_gemm")._reference_grouped_gemm

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case


MODE = "fwd"
HELION_KERNELS = (grouped_gemm_jagged_grid,)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    g, n = case.params["G"], case.params["N"]
    host = {
        f"a{i}": torch.rand(n, n, generator=gen, dtype=torch.bfloat16) for i in range(g)
    }
    host["b"] = torch.rand(n, n, generator=gen, dtype=torch.bfloat16)
    return host


def reference(*args: torch.Tensor) -> torch.Tensor:
    *groups, b = args
    b = b.to("mps", torch.float32)
    return torch.cat([a.to("mps", torch.float32) @ b for a in groups]).cpu()


def eager(*args: torch.Tensor) -> torch.Tensor:
    *groups, b = args
    return _reference_grouped_gemm(list(groups), [b] * len(groups))


def helion(*args: torch.Tensor) -> torch.Tensor:
    *groups, b = args
    starts = [0]
    for a in groups:
        starts.append(starts[-1] + a.size(0))
    # A blocking host-to-device copy would wait for all queued GPU work; the
    # offsets are only read by the kernel queued after it.
    offsets = torch.tensor(starts, dtype=torch.int32).to(b.device, non_blocking=True)
    return grouped_gemm_jagged_grid(torch.cat(groups), b, offsets)


def mlx(*args: mx.array) -> mx.array:
    import mlx.core as mx

    *groups, b = args
    return mx.concatenate(groups) @ b
