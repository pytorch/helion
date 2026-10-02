"""Pallas (JAX/TPU) backend — portable test registration and gap declarations.

All known gaps between the Pallas backend and the portable test suite are
declared here via ``xfail_test()``.  To fix a gap: implement the missing
feature, remove the ``xfail_test`` call, and verify with::

    HELION_BACKEND=pallas pytest test/portable/

Sub-variant axis
----------------
Within the Pallas backend, ``HELION_PALLAS_INTERPRET=1`` runs tests in CPU
simulation (interpret) mode rather than compiling to real Mosaic / TPU code.
Some gaps only exist on real TPU and some only in interpret mode — use the
``condition=`` parameter of ``xfail_test`` to restrict those to one path.
"""

from __future__ import annotations

from test.backends import register_backend
from test.backends import xfail_test

_backend = register_backend("pallas")


def _is_pallas_tpu() -> bool:
    """Return True when running on real TPU (Mosaic), not in interpret mode.

    Uses the same ``HELION_PALLAS_INTERPRET`` parsing as the framework's own
    ``helion.runtime.settings.is_pallas_interpret()`` (accepts "1", "true",
    "yes", "on" case-insensitively) so they stay in sync.
    """
    from helion.runtime.settings import is_pallas_interpret

    return not is_pallas_interpret()


def _is_pallas_interpret() -> bool:
    """Return True when running in Pallas interpret mode, not on real TPU."""
    from helion.runtime.settings import is_pallas_interpret

    return is_pallas_interpret()


# ---------------------------------------------------------------------------
# test_views
# ---------------------------------------------------------------------------

# torch.stack has no Pallas / Mosaic lowering yet.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_stack_power_of_2",
    reason="torch.stack not supported on pallas",
)
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_stack_non_power_of_2",
    reason="torch.stack not supported on pallas",
)
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_stack_dim0",
    reason="torch.stack not supported on pallas",
)

# Bitcast / dtype-reinterpret view not supported on Pallas.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_view_dtype_reinterpret",
    reason="view dtype reinterpret not supported on pallas",
)

# Mosaic (real-TPU compilation path) cannot reshape a 1-D vector to [..., 2].
# This passes in Pallas interpret mode (CPU simulation); restrict to real TPU.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_view_blocksize_constexpr_pairsum",
    reason="Mosaic does not support reshaping a 1D vector to [..., 2] on TPU",
    condition=_is_pallas_tpu(),
)

# JAX interpret-mode has a discharge bug on pipeline buffers; the real-TPU
# compilation path passes.  Restrict to the interpret path.
xfail_test(
    _backend,
    "test_views",
    "TestViews::test_reshape_input_types",
    reason="jax interpret-mode discharge bug on pipeline buffers",
    condition=_is_pallas_interpret(),
)
