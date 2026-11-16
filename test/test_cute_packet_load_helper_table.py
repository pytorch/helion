"""``PACKET_LOAD_HELPERS`` reproduces the helper names, width tables and
``CuteBackend.library_imports`` rows generated code bound before the table
existed.  The preamble emits a kernel's imports in ``library_imports`` order,
so the order of the rows is part of the rendered source and is pinned too.
"""

from __future__ import annotations

import importlib

import pytest

from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from helion._compiler.cute.cache_policy_loads import _CUTE_CACHE_LOAD_HELPERS
from helion._compiler.cute.cache_policy_loads import _CUTE_NC_L2_FIRST_HELPERS
from helion._compiler.cute.cache_policy_loads import MASKED_PACKET_LOAD_HELPER_NAMES
from helion._compiler.cute.cache_policy_loads import MASKED_PACKET_LOAD_HELPERS
from helion._compiler.cute.cache_policy_loads import PACKET_LOAD_HELPERS

# Table pins, not renders: they run on every backend row (the one test that
# imports the DSL modules skips without ``cutlass``).

# The literal ``library_imports`` rows the table replaced, in their order.
_LIBRARY_IMPORTS = (
    "from helion._compiler.cute.l2_policy import load_v16b_l2_evict_last as _cute_load_l2_evict_last",
    "from helion._compiler.cute.masked_packet_load import load_v64b_if as _cute_load_v64b_if",
    "from helion._compiler.cute.masked_packet_load import load_v32b_if as _cute_load_v32b_if",
    "from helion._compiler.cute.masked_packet_load import load_v16b_if as _cute_load_v16b_if",
    "from helion._compiler.cute.masked_packet_load import load_v8b_if as _cute_load_v8b_if",
    "from helion._compiler.cute.masked_packet_load import load_v4b_if as _cute_load_v4b_if",
    "from helion._compiler.cute.readonly_packet_load_helpers import load_nc_l2_evict_first_32b as _cute_load_nc_l2_evict_first_32b",
    "from helion._compiler.cute.readonly_packet_load_helpers import load_nc_l2_evict_first_16b as _cute_load_nc_l2_evict_first_16b",
    "from helion._compiler.cute.l2_policy import scalar_policy_load as _cute_scalar_policy_load",
    "from helion._compiler.cute.l2_policy import load_v16b_l1_l2_evict_first as _cute_load_l1_l2_evict_first",
    "from helion._compiler.cute.l2_policy import load_v8b_l1_l2_evict_first as _cute_load_l1_l2_evict_first_8b",
    "from helion._compiler.cute.l2_policy import load_v16b_l1_l2_evict_last as _cute_load_l1_l2_evict_last",
    "from helion._compiler.cute.l2_policy import load_v8b_l1_l2_evict_last as _cute_load_l1_l2_evict_last_8b",
    "from helion._compiler.cute.l2_policy import load_v8b_l2_evict_last as _cute_load_l2_evict_last_8b",
    "from helion._compiler.cute.l2_policy import load_v4b_l2_evict_last as _cute_load_l2_evict_last_4b",
    "from helion._compiler.cute.l2_policy import load_v4b_l1_l2_evict_first as _cute_load_l1_l2_evict_first_4b",
    "from helion._compiler.cute.l2_policy import load_v4b_l1_l2_evict_last as _cute_load_l1_l2_evict_last_4b",
    "from helion._compiler.cute.l2_policy import load_v32b_l2_evict_last as _cute_load_l2_evict_last_32b",
    "from helion._compiler.cute.l2_policy import load_v32b_l1_l2_evict_first as _cute_load_l1_l2_evict_first_32b",
    "from helion._compiler.cute.l2_policy import load_v32b_l1_l2_evict_last as _cute_load_l1_l2_evict_last_32b",
)
# The literal width tables the table replaced.
_CACHE_LOAD_HELPERS = {
    (32, "l2_last"): "_cute_load_l2_evict_last_32b",
    (32, "l1_l2_first"): "_cute_load_l1_l2_evict_first_32b",
    (32, "l1_l2_last"): "_cute_load_l1_l2_evict_last_32b",
    (16, "l2_last"): "_cute_load_l2_evict_last",
    (16, "l1_l2_first"): "_cute_load_l1_l2_evict_first",
    (16, "l1_l2_last"): "_cute_load_l1_l2_evict_last",
    (8, "l2_last"): "_cute_load_l2_evict_last_8b",
    (8, "l1_l2_first"): "_cute_load_l1_l2_evict_first_8b",
    (8, "l1_l2_last"): "_cute_load_l1_l2_evict_last_8b",
    (4, "l2_last"): "_cute_load_l2_evict_last_4b",
    (4, "l1_l2_first"): "_cute_load_l1_l2_evict_first_4b",
    (4, "l1_l2_last"): "_cute_load_l1_l2_evict_last_4b",
}
_NC_L2_FIRST_HELPERS = {
    32: "_cute_load_nc_l2_evict_first_32b",
    16: "_cute_load_nc_l2_evict_first_16b",
}
_MASKED_PACKET_LOAD_HELPERS = {
    64: "_cute_load_v64b_if",
    32: "_cute_load_v32b_if",
    16: "_cute_load_v16b_if",
    8: "_cute_load_v8b_if",
    4: "_cute_load_v4b_if",
}


def test_rows_are_the_literal_library_imports_in_order() -> None:
    assert [helper.import_line for helper in PACKET_LOAD_HELPERS] == list(
        _LIBRARY_IMPORTS
    )
    assert [helper.name for helper in PACKET_LOAD_HELPERS] == [
        line.rsplit(" as ", 1)[1] for line in _LIBRARY_IMPORTS
    ]


def test_backend_binds_the_rows_in_table_order() -> None:
    imports = CuteBackend().library_imports
    names = [helper.name for helper in PACKET_LOAD_HELPERS]
    assert [name for name in imports if name in names] == names
    assert [imports[name] for name in names] == list(_LIBRARY_IMPORTS)


def test_width_tables_are_the_literals() -> None:
    assert _CUTE_CACHE_LOAD_HELPERS == _CACHE_LOAD_HELPERS
    assert _CUTE_NC_L2_FIRST_HELPERS == _NC_L2_FIRST_HELPERS
    assert MASKED_PACKET_LOAD_HELPERS == _MASKED_PACKET_LOAD_HELPERS
    assert (
        frozenset(_MASKED_PACKET_LOAD_HELPERS.values())
        == MASKED_PACKET_LOAD_HELPER_NAMES
    )
    assert (
        frozenset(_CACHE_LOAD_HELPERS.values())
        | frozenset(_NC_L2_FIRST_HELPERS.values())
        == _CUTE_CACHE_LOAD_HELPER_NAMES
    )


def test_every_row_binds_a_helper_of_its_name() -> None:
    pytest.importorskip("cutlass")
    for helper in PACKET_LOAD_HELPERS:
        module = importlib.import_module(f"helion._compiler.cute.{helper.module}")
        function = getattr(module, helper.symbol)
        # The factory rows of ``l2_policy`` / ``masked_packet_load`` carry the
        # name of their binding, like a hand-written ``def``.
        assert function.__name__ == helper.symbol, helper
        scope: dict[str, object] = {}
        exec(helper.import_line, scope)
        assert scope[helper.name] is function
