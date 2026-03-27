"""
Grid Fission for Matrix Multiplication
=======================================

This example explains **grid fission**, a Helion compiler feature that controls
how tile dimensions map to the GPU launch grid versus inner device loops.

What is Grid Fission?
---------------------

In a standard tiled matmul the outer ``hl.tile([M, N])`` creates a 2-D launch
grid: one thread-block per (M-tile, N-tile) pair.  Grid fission lets you
*remove* some of those dimensions from the launch grid and iterate over them
inside each thread-block instead.

Consider ``M = N = 8192`` with ``block_size = 128``.  Without fission we launch
``64 x 64 = 4096`` thread-blocks.  With partial fission on M (factor=4) we
launch ``16 x 64 = 1024`` blocks, each looping over 4 consecutive M-tiles::

  No fission  [0, 0]                  Partial fission on M, factor=4  [4, 0]
  ─────────────────────               ──────────────────────────────────────
  Launch grid: 64 x 64 = 4096 blocks  Launch grid: 16 x 64 = 1024 blocks
                                       Each block handles 4 consecutive M-tiles

  GPU grid (each cell = 1 block)       GPU grid          Inner loop (per block)
  ┌────┬────┬────┬─··─┬────┐           ┌────┬────┬─··─┬────┐
  │0,0 │0,1 │0,2 │    │0,63│           │0,0 │0,1 │    │0,63│  block (0,j) →
  ├────┼────┼────┼─··─┼────┤           ├────┼────┼─··─┼────┤  M-tiles 0,1,2,3
  │1,0 │1,1 │1,2 │    │1,63│           │1,0 │1,1 │    │1,63│  block (1,j) →
  ├────┼────┼────┼─··─┼────┤           ├────┼────┼─··─┼────┤  M-tiles 4,5,6,7
  │ ·  │ ·  │ ·  │    │ ·  │           │ ·  │ ·  │    │ ·  │
  ├────┼────┼────┼─··─┼────┤           ├────┼────┼─··─┼────┤
  │63,0│    │    │    │63,63│          │15,0│    │    │15,63│  block (15,j) →
  └────┴────┴────┴─··─┴────┘           └────┴────┴─··─┴────┘  M-tiles 60..63

With full fission on N ``[0, -1]`` we launch only ``64`` blocks, each looping
over all 64 N-tiles internally::

  Full fission on N  [0, -1]
  ──────────────────────────
  Launch grid: 64 x 1 = 64 blocks

  GPU grid         Inner loop
  ┌────┐  ─┐
  │ 0  │   │ each block loops
  ├────┤   │ over all N-tiles
  │ 1  │   │ 0, 1, 2, ..., 63
  ├────┤   │
  │ ·  │   │
  ├────┤   │
  │ 63 │   │
  └────┘  ─┘

Why use grid fission?
~~~~~~~~~~~~~~~~~~~~~

- **L2 cache reuse**: when one block processes multiple M-tiles sequentially,
  the right-matrix columns ``y[tile_k, tile_n]`` stay in L2 cache and are
  reused across iterations -- this is the main performance win.
- **Reduced grid overhead**: fewer blocks means less scheduling overhead, which
  matters when the blocks themselves are fast (small tiles or small problem).
- **Autotuner flexibility**: the autotuner can search over fission factors to
  find the sweet spot between parallelism and locality for each problem size.

``grid_fissions`` API
~~~~~~~~~~~~~~~~~~~~~

``grid_fissions`` is set in ``helion.Config`` as a list-of-lists, one inner list
per ``hl.tile()`` call that has 2+ dimensions.  Each integer in the inner list
is a **fission factor** for the corresponding tile dimension:

  - ``0``  -- no fission (dimension stays fully in the launch grid)
  - ``-1`` -- full fission (dimension removed from grid, becomes inner loop)
  - ``k``  -- partial fission by factor *k* (grid shrinks by *k*; valid: 2,4,8,16,32,64)

The kernel code itself is **identical** across all variants -- only the
``Config`` changes.  This is a key design feature: the autotuner can explore
fission strategies without touching the kernel source.  This example demonstrates
this by defining a **single kernel** and using ``bind()`` / ``compile_config()``
to create multiple variants with different fission settings.

Compatibility with Other Helion Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grid fission interacts with several other Helion features:

- **Persistent PID types** (``persistent_blocked``, ``persistent_interleaved``):
  **Incompatible** with grid fission.  Persistent kernels manage their own grid
  traversal, so fission factors are forced to all-zeros.  The autotuner silently
  zeroes them out; manually combining both raises ``InvalidConfig``.

- **``xyz`` PID type**: Compatible, but if fission removes enough grid dimensions
  that only 0 or 1 remain, Helion silently falls back to ``flat`` PID type.

- **``flat`` PID type**: Fully compatible, no restrictions.

- **Loop flattening** (``flatten_loops``): Grid fission **overrides** flattening.
  When any fission factor is non-zero the compiler selects ``NDTileStrategy``
  instead of ``FlattenedTileStrategy``, because fission requires per-dimension
  PID and loop handling.  No error is raised -- fission simply takes priority.

::

  ┌─────────────────────┬──────────────────────────────────────┐
  │ Feature             │ Compatibility with grid_fissions     │
  ├─────────────────────┼──────────────────────────────────────┤
  │ pid_type="flat"     │ Fully compatible                     │
  │ pid_type="xyz"      │ Compatible; falls back to flat if    │
  │                     │ effective grid dims <= 1              │
  │ pid_type=           │ Incompatible -- fission forced to    │
  │   "persistent_*"    │ all-zeros or InvalidConfig raised    │
  │ flatten_loops=True  │ Overridden when any fission != 0     │
  └─────────────────────┴──────────────────────────────────────┘

Benchmark Results (H100 80GB, M=N=8192, K=2048)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  =================================================================
  Implementation       Time (ms)    Speedup (vs torch.matmul)
  -----------------------------------------------------------------
  partial_M=4 [4,0]    0.4917       0.75x          <-- best helion
  partial_M=2 [2,0]    0.5169       0.71x
  no_fission  [0,0]    0.5626       0.66x
  full_N      [0,-1]   1.6362       0.23x
  torch.matmul         0.3694       1.00x (ref)
  =================================================================

Key takeaways:

- Partial fission on M (factor=4) is **13% faster** than no fission because
  each thread block processes 4 consecutive 128-row strips, reusing the same
  ``y[tile_k, tile_n]`` columns from L2 cache across iterations.
- Increasing the fission factor beyond the L2-reuse sweet spot (full fission
  on N) destroys parallelism and is **3x slower**.
- ``torch.matmul`` (cuBLAS) is still faster overall due to hand-tuned assembly;
  this example shows the relative benefit of fission within the Helion kernel
  space.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# %%
# Kernel Definition
# -----------------
# A single kernel function handles all fission variants.  The ``Config`` passed
# at bind/compile time is the only thing that changes the launch grid strategy.


# %%
@helion.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tiled matrix multiplication: out = x @ y."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


# %%
# Fission Configs
# ---------------
# Each config uses the same block sizes / warps / stages but varies
# ``grid_fissions`` to change how the M and N tile dimensions are mapped
# to the GPU launch grid versus inner device loops.
CONFIGS = {
    "no_fission [0,0]": helion.Config(
        block_sizes=[128, 128, 32],
        grid_fissions=[[0, 0]],
        num_warps=8,
        num_stages=3,
    ),
    "partial_M=2 [2,0]": helion.Config(
        block_sizes=[128, 128, 32],
        grid_fissions=[[2, 0]],
        num_warps=8,
        num_stages=3,
    ),
    "partial_M=4 [4,0]": helion.Config(
        block_sizes=[128, 128, 32],
        grid_fissions=[[4, 0]],
        num_warps=8,
        num_stages=3,
    ),
    "full_N [0,-1]": helion.Config(
        block_sizes=[128, 128, 32],
        grid_fissions=[[0, -1]],
        num_warps=8,
        num_stages=3,
    ),
}


# %%
# Verification and Benchmarking
# -----------------------------
# We ``bind()`` the kernel once for the given input shapes, then
# ``compile_config()`` each fission variant.  The resulting compiled
# runners are plain callables that ``run_example`` can benchmark.


# %%
def check(m: int, k: int, n: int) -> None:
    x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

    bound = matmul.bind((x, y))
    kernels = {
        name: bound.compile_config(cfg) for name, cfg in CONFIGS.items()
    }
    run_example(kernels, torch.matmul, (x, y))


# %%
# Main
# ----


# %%
def main() -> None:
    # Large M and N with moderate K: partial fission on M improves L2 reuse
    # of the right matrix y[tile_k, tile_n] across consecutive M-blocks.
    check(8192, 2048, 8192)


if __name__ == "__main__":
    main()
