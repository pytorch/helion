"""Apple hardware limits and the tensor dtypes shared by the Metal backend.

Kept in a leaf module (no package imports) so codegen (``backend.py``,
``reduction.py``, ``mpp_matmul.py``, ``masked_branch.py``), the MSL preamble
(``msl_reduction.py``) and ``metal_jit.py`` can all use these without import
cycles.
"""

from __future__ import annotations

import torch
from torch._inductor.codegen.mps import DTYPE_TO_METAL as _INDUCTOR_DTYPE_TO_METAL

# Metal SIMD-group width on all Apple GPUs.  ``c10/metal/reduction_utils.h``
# hardcodes the same assumption (see its "simdgroup is 32" comment).
SIMD_WIDTH = 32

#: Maximum threads in a Metal threadgroup.
#:
#: Numerically equal to ``per_thread_planning.MAX_THREADS_PER_BLOCK``, the cap
#: the shared per-thread tile planner (``plan_per_thread_loop_strategy``)
#: enforces; everything on this backend uses this name.
MAX_THREADS_PER_THREADGROUP = 1024

#: Threadgroup memory available to one threadgroup
#: (``maxThreadgroupMemoryLength`` on Apple-silicon GPUs).
MAX_THREADGROUP_MEMORY_BYTES = 32768

#: A Metal threadgroup is three-dimensional:
#: ``thread_position_in_threadgroup`` is a ``uint3``.  A kernel needing a
#: fourth thread axis has no way to express it, and indexing ``tid[3]``
#: reads past the end of that vector -- which the shader compiler accepts
#: and which silently returns garbage.
MAX_THREAD_AXES = 3

#: The dtypes whose tensors Helion's MSL binds, loads and stores.
_SUPPORTED_DTYPES = frozenset(
    {
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    }
)
#: The tensor dtypes the Metal backend supports, with the MSL type each one
#: binds and loads as: Inductor's ``DTYPE_TO_METAL`` restricted to the dtypes
#: Helion's MSL handles, so the set is what each PyTorch version's table
#: offers of those (2.9 to 2.11 have no ``uint16``, ``uint32`` or
#: ``uint64``).  Since PyTorch 2.16 Inductor also maps the float8 types to
#: ``c10::metal::float8_*`` and ``float4_e2m1fn_x2`` to ``uchar``; Helion
#: handles none of them (a masked fp8 load converts to and from ``float``
#: ambiguously), so tensor arguments and ``MetalBackend.dtype_str`` reject
#: them.  An in-kernel cast still lowers through Inductor's
#: ``MetalOverrides.to_dtype``, which reads Inductor's table.
DTYPE_TO_METAL: dict[torch.dtype, str] = {
    dtype: name
    for dtype, name in _INDUCTOR_DTYPE_TO_METAL.items()
    if dtype in _SUPPORTED_DTYPES
}
