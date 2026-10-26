"""Regression tests for write-attribution and PDL-dominance holes found by the canon5 review.

Each fixture under test/data/ttgir_prefetch_findings is the saved sm_100a TTGIR of a GPU reproducer that raced
(40/40 bad launches) when the pass hoisted copies of a tensor the kernel itself writes, or hoisted above a wait
that does not dominate the copies.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from helion._testing import TestCase
from helion.runtime.triton import ttgir_weight_prefetch as prefetch

_FIXTURES = Path(__file__).parent / "data" / "ttgir_prefetch_findings"


def _hoisted(name: str) -> int:
    return prefetch.rewrite((_FIXTURES / f"{name}.ttgir").read_text())[1].hoisted


class TestPrefetchFindings(TestCase):
    def test_stamp_through_an_integer_argument_writes_its_target(self) -> None:
        # the allowlisted stamp stores through `xaddr`, an i64 kernel argument holding x's address
        self.assertEqual(_hoisted("ia"), 0)

    def test_callee_stamp_through_a_loaded_address_writes_every_argument(self) -> None:
        # noinline callee: addr = load(table); stamp through addr
        self.assertEqual(_hoisted("cxs"), 0)

    def test_callee_int_to_ptr_store_writes_every_argument(self) -> None:
        # noinline callee: tt.store through int_to_ptr(load(table))
        self.assertEqual(_hoisted("cxi"), 0)

    def test_wait_in_a_sibling_block_does_not_dominate(self) -> None:
        # top-level `if pid == 100000: gdc_wait(); return` gives ^bb1 (wait, return) / ^bb2 (spin, copies)
        self.assertEqual(_hoisted("mb"), 0)

    def test_store_through_pointer_plus_distance_to_another_argument(self) -> None:
        # tt.store(a + (x - a) + offs) writes x
        self.assertEqual(_hoisted("pd"), 0)

    def test_stamp_through_pointer_plus_distance_to_another_argument(self) -> None:
        self.assertEqual(_hoisted("pda"), 0)

    def test_controls_keep_their_hoists(self) -> None:
        self.assertEqual(
            _hoisted("triton_top"), 2
        )  # exact gdc_wait at the top: a legitimate hoist
        self.assertEqual(
            _hoisted("nowait"), 2
        )  # the module never waits on its predecessor
        for name in ("nb", "pdx", "pdax", "iax", "cxx"):
            self.assertEqual(_hoisted(name), 0, name)


if __name__ == "__main__":
    unittest.main()
