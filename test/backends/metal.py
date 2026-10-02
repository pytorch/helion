"""Apple Metal (MSL) backend — portable test registration and gap declarations.

Gaps were triaged by running ``HELION_BACKEND=metal pytest test/portable/`` on
Apple Silicon hardware.  Add new ``xfail_test()`` calls here when new portable
tests are introduced that fail on Metal.
"""

from __future__ import annotations

from test.backends import register_backend
from test.backends import xfail_test

_backend = register_backend("metal")

# ---------------------------------------------------------------------------
# test_views
# ---------------------------------------------------------------------------

# hl.split / hl.join have no Metal codegen yet.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_split_join_roundtrip",
    reason="hl.split not implemented on the Metal backend",
)
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_join_broadcast_scalar",
    reason="hl.join not implemented on the Metal backend",
)

# aten.view / reshape not lowered on Metal.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_softmax_view_reshape",
    reason="aten.view not lowered on the Metal backend",
)
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_reshape_input_types",
    reason="aten.view not lowered on the Metal backend",
)
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_view_blocksize_constexpr_pairsum",
    reason="aten.view not lowered on the Metal backend",
)

# aten.view (dtype reinterpret / bitcast) not lowered on Metal.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_view_dtype_reinterpret",
    reason="aten.view dtype reinterpret not lowered on the Metal backend",
)

# aten.stack not lowered on Metal.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_stack_power_of_2",
    reason="aten.stack not lowered on the Metal backend",
)

# aten.permute not lowered on Metal.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_transpose_T_unsqueeze",
    reason="aten.permute not lowered on the Metal backend",
)

# Metal thread-count limit prevents persistent reductions over large/non-power-of-2
# element counts (max 1024 threads).
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_reshape_sum",
    reason="persistent reduction exceeds Metal max thread count (1024)",
)
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_stack_non_power_of_2",
    reason="persistent reduction exceeds Metal max thread count (1024)",
)
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_stack_dim0",
    reason="persistent reduction exceeds Metal max thread count (1024)",
)
