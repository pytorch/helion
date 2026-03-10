"""
Softmax with Advanced Controls File (ACF)
==========================================

This example demonstrates how to use ``advanced_controls_file`` in ``helion.Config``
and ``autotune_search_acf`` to pass PTXAS register/scheduling hints to Helion kernels.

Helion provides two ways to use ACFs:

- ``advanced_controls_file`` in ``helion.Config``
- ``autotune_search_acf``

The ``advanced_controls_file`` parameter is used to specify a single ACF file to be applied to a kernel.
The ``autotune_search_acf`` parameter is used to specify a list of ACF files to be considered during autotuning.

The kernel body is defined once as a plain function and four named variants are
created from it by calling ``helion.kernel(fn, ...)`` with different settings:

- **softmax_default** — baseline, no ACF, autotuner picks freely.
- **softmax_tune_acf** — autotuner searches over the provided ACFs (plus ``""``
  which is appended automatically, so the no-ACF baseline is always considered).
- **softmax_example_acf** — full config pre-pinned with the well-tuned ACF;
  no autotuning at call time.
- **softmax_noopt_acf** — full config pre-pinned with the no-optimization ACF;
  shows worst-case performance.

.. note::
   The ACFs will be available for download in the near future.

.. note::
   This feature is still highly experimental. It could cause incorrect results or runtime errors if ACFs are used with the wrong hardware or use case.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from pathlib import Path

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

_ACF_DIR = Path(__file__).parent
_ACF_EXAMPLE = "h100_example1.acf"
_ACF_NOOPT = "noopt.acf"
_ACFS_PATHS = dict.fromkeys([_ACF_EXAMPLE, _ACF_NOOPT], "")

for acf_name in _ACFS_PATHS:
    acf_path = _ACF_DIR / acf_name
    if acf_path.exists():
        _ACFS_PATHS[acf_name] = str(acf_path)
    else:
        print("=" * 65)
        print(
            f"ACF file not found:\n    {acf_path}\nRunning with default profile instead."
        )
        print("=" * 65)
        _ACFS_PATHS[acf_name] = ""

# %%
# Kernel Body
# -----------
# The softmax logic is defined exactly once as a plain function.
# All variants below reuse this body via ``helion.kernel(fn, ...)``.


# %%
def _softmax(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# %%
# Variant 1 — Baseline (no ACF)
# ------------------------------

# %%
softmax_default = helion.kernel()(_softmax)

# %%
# Variant 2 — Autotune over ACFs
# --------------------------------
# The autotuner searches over the supplied ACFs.  An empty string ``""`` is
# appended automatically so the no-ACF baseline is always included in the search.


# %%
softmax_tune_acf = helion.kernel(
    autotune_compile_timeout=120,
    autotune_search_acf=list(_ACFS_PATHS.values()),
    static_shapes=True,
)(_softmax)

# %%
# Variant 3 — Pre-pinned well-tuned ACF
# ----------------------------------------
# Full config specified upfront — no autotuning at call time.
# Use this once you have identified the best config for your workload and hardware.
#
# .. note::
#    ``h100_example1.acf`` and the block/indexing/stage/warp values below were
#    tuned specifically for **H100 (SM90)**. You will need a different ACF and
#    configuration when using a different GPU architecture (e.g. B200/SM100 or
#    A100/SM80); using mismatched ACFs can cause incorrect
#    results or runtime errors.


# %%
softmax_example_acf = helion.kernel(
    config=helion.Config(
        advanced_controls_file=_ACFS_PATHS[_ACF_EXAMPLE],
        block_sizes=[4],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        num_stages=2,
        num_warps=8,
        reduction_loops=[512],
    ),
    static_shapes=True,
)(_softmax)

# %%
# Variant 4 — Pre-pinned no-optimization ACF
# --------------------------------------------
# Serves as a worst-case contrast against ``softmax_example_acf``.
# The block/indexing config is also H100 (SM90)-specific; see the note above.


# %%
softmax_noopt_acf = helion.kernel(
    config=helion.Config(
        advanced_controls_file=_ACFS_PATHS[_ACF_NOOPT],
        block_sizes=[4],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        num_stages=2,
        num_warps=8,
        reduction_loops=[512],
    ),
    static_shapes=True,
)(_softmax)


# %%
# Verification
# ------------
# You should expect an output similar to this::
#
#    =================================================================
#    Benchmark Results
#    =================================================================
#    Implementation       Time (ms)    Speedup
#    -----------------------------------------------------------------
#    helion default       0.0284       2.16x
#    helion tune acf      0.0281       2.18x
#    helion example acf   0.0359       1.70x
#    helion noopt acf     0.0785       0.78x
#    torch                0.0612       1.00x (ref)
#    =================================================================


# %%
def check(m: int, n: int) -> None:
    """
    Run correctness checks comparing all ACF variants against PyTorch softmax.
    Args:
        m (int): Number of rows in input tensor.
        n (int): Number of columns in input tensor.
    """
    x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
    kernels = {
        "helion default": softmax_default,
        "helion tune acf": softmax_tune_acf,
        "helion example acf": softmax_example_acf,
        "helion noopt acf": softmax_noopt_acf,
    }
    run_example(kernels, lambda x: torch.nn.functional.softmax(x, dim=1), (x,))

    if "" in _ACFS_PATHS.values():
        print("=" * 65)
        print("Some ACF files were not found")
        print(
            "the values in the table above are not representative of actual behavior."
        )
        print("ACFs will be available for download in the near future.")
        print("=" * 65)


# %%
# Main
# ----


# %%
def main() -> None:
    """Run the softmax ACF example with a representative input size."""
    check(4096, 2560)


if __name__ == "__main__":
    main()
