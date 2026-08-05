"""Apple hardware limits shared by the Metal backend.

Kept in a leaf module (no package imports) so codegen (``backend.py``,
``reduction.py``) and the MSL preamble (``msl_reduction.py``) can all use
these without import cycles.
"""

from __future__ import annotations

# Metal SIMD-group width on all Apple GPUs.  ``c10/metal/reduction_utils.h``
# hardcodes the same assumption (see its "simdgroup is 32" comment).
SIMD_WIDTH = 32

#: Maximum threads in a Metal threadgroup.
#:
#: Numerically equal to ``cute.thread_budget.MAX_THREADS_PER_BLOCK``: one
#: Apple hardware limit under two names.  The MPP budget in ``backend.py``
#: uses the CuTe name; everything else on this backend uses this one.
MAX_THREADS_PER_THREADGROUP = 1024

#: Maximum number of SIMD groups in a threadgroup, hence the slot count of
#: every ``threadgroup`` scratch buffer (one slot per SIMD group; see
#: ``reduction.alloc_threadgroup_buffer``).
MAX_SIMD_GROUPS = MAX_THREADS_PER_THREADGROUP // SIMD_WIDTH

#: A Metal threadgroup is three-dimensional:
#: ``thread_position_in_threadgroup`` is a ``uint3``.  A kernel needing a
#: fourth thread axis has no way to express it, and indexing ``tid[3]``
#: reads past the end of that vector -- which the shader compiler accepts
#: and which silently returns garbage.
MAX_THREAD_AXES = 3
