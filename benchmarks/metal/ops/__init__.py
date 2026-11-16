"""One module per dashboard kernel.

Each module defines the same math four ways, all taking the cell's inputs
positionally in the order ``inputs`` returns them:

- ``inputs(case, generator)``: the host (CPU) input tensors, in the dtype the
  operator runs at.
- ``reference(*inputs)``: the oracle, from the host inputs.  It computes in
  float64 on the CPU unless its docstring says otherwise.
- ``eager(*inputs)``: the PyTorch baseline the kernel's example in
  ``examples/`` checks Helion against; also what ``torch_compile`` compiles.
- ``helion(*inputs)``: the Helion example the dashboard runs.
- ``mlx(*inputs)``: the strongest idiomatic MLX formulation, run uncompiled.

``MODE`` is ``"fwd"`` or ``"bwd"``.  A backward operator's ``eager``,
``helion`` and ``mlx`` are forwards; the harness differentiates them with
respect to ``GRAD_WRT`` for the cotangent input ``dy``, and its ``reference``
returns the gradients directly.  ``HELION_KERNELS`` lists the Helion kernels
``helion`` runs.

Deliberate differences from the dashboard: forward inputs do not require
grad (TritonBench's do, which adds autograd bookkeeping to eager calls);
``eager`` is the example's baseline rather than TritonBench's (e.g. softmax's
is ``F.softmax`` where TritonBench's ``naive_softmax`` is hand-written); and
every kernel uses the same timing budgets, where ``benchmarks/run.py`` raises
gemm's ``--rep`` to 3000 ms; and grouped_gemm's ``helion`` runs a kernel of
this package (``_grouped_gemm_grid``) rather than the example's, so that its
config chooses the launch (see ``grouped_gemm.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Row chunk for host references, bounding their float64 temporaries.
REFERENCE_ROWS = 1024


def f64(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.double() for t in tensors)
